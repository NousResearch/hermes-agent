"""
Approval Delegation Plugin

Routes dangerous-command approvals to designated admins when the user
is not an admin. Supports cross-platform delegation.
"""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# i18n helper with fallback
_t = None
try:
    from agent.i18n import t as _t
except ImportError:
    logger.debug("[approval-delegation] i18n not available, using fallback")

# Platform enum (lazy import for testability)
Platform = None


def _ensure_platform_import():
    """Lazy import Platform enum."""
    global Platform
    if Platform is None:
        try:
            from gateway.config import Platform as P
            Platform = P
        except ImportError:
            logger.warning("[approval-delegation] Cannot import Platform enum")


def _tr(key: str, default: str, **kwargs) -> str:
    """Translate with fallback. Uses i18n if available, otherwise returns default.
    
    Returns fallback if:
    - i18n is not available (_t is None)
    - i18n raises an exception
    - i18n returns the key itself (untranslated)
    - i18n returns None
    """
    if _t is not None:
        try:
            result = _t(f"approval_delegation.{key}", **kwargs)
            # Use fallback if result is None, empty, or same as key (untranslated)
            if result and result != f"approval_delegation.{key}":
                return result
        except Exception:
            pass
    return default.format(**kwargs) if kwargs else default


# Thread-safe runner map: session_key -> (runner, timestamp)
# Replaces the unsafe _current_runner global variable
_runner_map: OrderedDict[str, tuple] = {}
_runner_map_lock = threading.Lock()
_RUNNER_TTL = 600  # 10 minutes

# Captured main event loop reference. Worker-thread callbacks (sync context)
# need this to schedule coroutines back onto the gateway's running loop,
# because asyncio.get_event_loop() raises in worker threads on Python 3.10+.
_main_loop: Optional[asyncio.AbstractEventLoop] = None
_main_loop_lock = threading.Lock()


def _capture_main_loop() -> None:
    """Capture the currently running event loop. Call from async context."""
    global _main_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    with _main_loop_lock:
        if _main_loop is None or _main_loop.is_closed() or _main_loop is not loop:
            _main_loop = loop


def _get_main_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Return the captured main event loop, or None if unavailable."""
    with _main_loop_lock:
        if _main_loop is not None and not _main_loop.is_closed():
            return _main_loop
    return None


def _store_runner(session_key: str, runner) -> None:
    """Store runner reference for a session_key."""
    if not session_key:
        return
    with _runner_map_lock:
        _runner_map[session_key] = (runner, time.monotonic())
        _cleanup_stale_runners()


def _get_runner(session_key: str):
    """Get runner for a session_key. Returns None if expired or missing."""
    if not session_key:
        return None
    with _runner_map_lock:
        entry = _runner_map.get(session_key)
        if entry is None:
            return None
        runner, ts = entry
        if (time.monotonic() - ts) > _RUNNER_TTL:
            _runner_map.pop(session_key, None)
            return None
        return runner


def _remove_runner(session_key: str) -> None:
    """Remove runner reference for a session_key."""
    if not session_key:
        return
    with _runner_map_lock:
        _runner_map.pop(session_key, None)


def _cleanup_stale_runners() -> None:
    """Clean up expired runner entries (caller must hold lock)."""
    now = time.monotonic()
    expired = [k for k, (_, ts) in _runner_map.items() if (now - ts) > _RUNNER_TTL]
    for k in expired:
        _runner_map.pop(k, None)


def _get_adapter(runner, platform_str: str):
    """Get platform adapter from runner. Returns None if not found.
    
    Args:
        runner: GatewayRunner instance
        platform_str: Platform string (e.g., 'feishu', 'weixin')
    
    Returns:
        Platform adapter or None
    """
    if not runner or not platform_str:
        return None
    
    _ensure_platform_import()
    if Platform is None:
        return None
    
    try:
        platform_enum = Platform(platform_str)
        return runner.adapters.get(platform_enum)
    except (ValueError, AttributeError) as e:
        logger.warning("[approval-delegation] Invalid platform '%s': %s", platform_str, e)
        return None


async def _send_message_safe(adapter, chat_id: str, message: str, metadata=None) -> bool:
    """Send message with proper async handling and error recovery.
    
    Args:
        adapter: Platform adapter
        chat_id: Target chat ID
        message: Message text
        metadata: Optional metadata dict
    
    Returns:
        True if sent successfully, False otherwise
    """
    try:
        await adapter.send(chat_id, message, metadata=metadata)
        return True
    except Exception as e:
        logger.warning("[approval-delegation] Failed to send message to %s: %s", chat_id[:16], e)
        return False


def _send_message_from_sync(adapter, chat_id: str, message: str, metadata=None, timeout: int = 10) -> bool:
    """Send message from synchronous context (worker thread) with timeout.
    
    Uses the captured main event loop to schedule the async send. Worker
    threads in Python 3.10+ cannot use asyncio.get_event_loop() — it raises
    when no loop is set on the current thread. We rely on _capture_main_loop()
    being called from an async context first (during _patch_run_agent).
    
    Args:
        adapter: Platform adapter
        chat_id: Target chat ID
        message: Message text
        metadata: Optional metadata dict
        timeout: Timeout in seconds
    
    Returns:
        True if sent successfully, False otherwise
    """
    main_loop = _get_main_loop()
    if main_loop is None:
        logger.warning(
            "[approval-delegation] No captured main loop; cannot send message to %s",
            chat_id[:16] if chat_id else "?",
        )
        return False
    
    try:
        # Schedule on the main loop from this worker thread
        fut = asyncio.run_coroutine_threadsafe(
            _send_message_safe(adapter, chat_id, message, metadata),
            main_loop,
        )
        return fut.result(timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "[approval-delegation] Send timed out (%ds) for %s",
            timeout, chat_id[:16] if chat_id else "?",
        )
        return False
    except Exception as e:
        logger.warning(
            "[approval-delegation] Failed to send message to %s: %s",
            chat_id[:16] if chat_id else "?", e,
        )
        return False


def _run_async_from_sync(coro, timeout: int = 15):
    """Run an async coroutine from synchronous context (worker thread).
    
    Similar to _send_message_from_sync but returns the coroutine's result
    instead of bool. Used for calling adapter methods that return SendResult.
    """
    main_loop = _get_main_loop()
    if main_loop is None:
        logger.warning("[approval-delegation] No captured main loop for async call")
        return None
    
    try:
        fut = asyncio.run_coroutine_threadsafe(coro, main_loop)
        return fut.result(timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("[approval-delegation] Async call timed out (%ds)", timeout)
        return None
    except Exception as e:
        logger.warning("[approval-delegation] Async call failed: %s", e)
        return None


def register(ctx: Optional[Dict] = None) -> None:
    """Plugin entry point — monkey-patch approval flow with delegation logic."""
    try:
        _patch_run_agent()
        _patch_register_gateway_notify()
        _patch_approve_deny_handlers()
        logger.info("[approval-delegation] Plugin loaded successfully")
    except Exception as e:
        logger.error("[approval-delegation] Failed to load: %s", e)
        import traceback
        traceback.print_exc()


def _patch_run_agent() -> None:
    """Patch _run_agent to store delegation context."""
    import gateway.run as run_module
    
    # Store original method
    _original_run_agent = run_module.GatewayRunner._run_agent
    
    async def _patched_run_agent(
        self,
        message: str,
        context_prompt: str,
        history: list,
        source,
        session_id: str,
        session_key: str = None,
        run_generation: Optional[int] = None,
        _interrupt_depth: int = 0,
        event_message_id: Optional[str] = None,
        channel_prompt: Optional[str] = None,
    ):
        """Patched _run_agent that stores delegation context."""
        # Capture the main event loop on first call (we are in async context here)
        _capture_main_loop()
        
        # Store runner in thread-safe map instead of global variable
        _store_runner(session_key, self)
        
        from gateway.approval_delegation.delegation import (
            get_delegation_admins, is_admin_user,
        )
        
        # Get user info from source
        _src_platform = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        _src_user_id = str(source.user_id or "")
        _src_chat_id = str(source.chat_id or "")
        _src_user_name = str(source.user_name or source.user_id or "unknown")
        
        # Check if delegation is configured
        _admins = get_delegation_admins()
        _is_admin = is_admin_user(_src_platform, _src_user_id)
        _should_delegate = bool(_admins and not _is_admin)
        
        # Store delegation context on the runner instance
        self._delegation_context = {
            "should_delegate": _should_delegate,
            "admins": _admins,
            "src_platform": _src_platform,
            "src_user_id": _src_user_id,
            "src_chat_id": _src_chat_id,
            "src_user_name": _src_user_name,
            "src_chat_meta": getattr(source, "thread_metadata", None),
        }
        
        logger.info(
            "[approval-delegation] Context: user=%s:%s, is_admin=%s, should_delegate=%s, session=%s",
            _src_platform, _src_user_id, _is_admin, _should_delegate,
            session_key[:16] if session_key else "?",
        )
        
        # Call original method
        try:
            return await _original_run_agent(
                self, message, context_prompt, history, source,
                session_id, session_key, run_generation,
                _interrupt_depth, event_message_id, channel_prompt,
            )
        finally:
            # Optional: clean up after request completes
            # _remove_runner(session_key)  # Uncomment if you want immediate cleanup
            pass
    
    # Apply patch
    run_module.GatewayRunner._run_agent = _patched_run_agent
    logger.info("[approval-delegation] Patched _run_agent")


def _patch_register_gateway_notify() -> None:
    """Patch register_gateway_notify to intercept approval notifications."""
    import tools.approval as approval_module
    
    # Store original function
    _original_register = approval_module.register_gateway_notify
    
    def _patched_register_gateway_notify(session_key, callback):
        """Wrapped version that intercepts approval notifications for delegation."""
        
        def _delegated_callback(approval_data: dict) -> None:
            """Intercept approval notification and delegate if needed."""
            # Get runner from thread-safe map using session_key
            runner = _get_runner(session_key)
            
            if runner is None:
                logger.warning(
                    "[approval-delegation] No runner found for session %s, calling original callback",
                    session_key[:16] if session_key else "?",
                )
                callback(approval_data)
                return
            
            # Get delegation context from the runner
            ctx = getattr(runner, "_delegation_context", {})
            should_delegate = ctx.get("should_delegate", False)
            
            logger.info(
                "[approval-delegation] Callback: should_delegate=%s, session_key=%s",
                should_delegate, session_key[:16] if session_key else "?",
            )
            
            if not should_delegate:
                # User is admin or no delegation configured
                callback(approval_data)
                return
            
            # Non-admin user with delegation configured
            admins = ctx.get("admins", [])
            src_platform = ctx.get("src_platform", "")
            src_chat_id = ctx.get("src_chat_id", "")
            src_user_name = ctx.get("src_user_name", "unknown")
            src_chat_meta = ctx.get("src_chat_meta")
            
            # Get approval info
            cmd = approval_data.get("command", "")
            desc = approval_data.get("description", "") or _tr("dangerous_command", "dangerous command")
            
            logger.info(
                "[approval-delegation] Delegating approval: user=%s:%s, cmd=%s",
                src_platform, src_user_name, cmd[:50],
            )
            
            # Send "waiting for admin" message to user
            user_adapter = _get_adapter(runner, src_platform)
            if user_adapter:
                user_msg = _tr(
                    "waiting_for_admin",
                    "⏳ Approval required. Notifying admin...\n> {desc}",
                    desc=desc,
                )
                user_adapter.pause_typing_for_chat(src_chat_id)
                _send_message_from_sync(
                    user_adapter, src_chat_id, user_msg, metadata=src_chat_meta
                )
            
            # Send approval request to each admin
            from gateway.approval_delegation.delegation import register_delegation
            
            successful_admins = 0
            for admin in admins:
                admin_platform = admin["platform"]
                admin_chat = str(admin["chat_id"] or admin["user_id"])
                
                # Get the admin's platform adapter
                admin_adapter = _get_adapter(runner, admin_platform)
                if not admin_adapter:
                    logger.warning(
                        "[approval-delegation] No adapter for admin platform %s",
                        admin_platform,
                    )
                    continue
                
                # Register delegation
                register_delegation(
                    admin_platform=admin_platform,
                    admin_chat_id=admin_chat,
                    target_session_key=session_key,
                    user_platform=src_platform,
                    user_chat_id=src_chat_id,
                    user_chat_meta=src_chat_meta,
                )
                
                # Send approval request to admin
                # Try interactive card with buttons first (Feishu supports this)
                cmd_preview = cmd[:200] + "..." if len(cmd) > 200 else cmd
                sent = False
                if hasattr(admin_adapter, 'send_exec_approval'):
                    try:
                        result = _run_async_from_sync(
                            admin_adapter.send_exec_approval(
                                chat_id=admin_chat,
                                command=cmd,
                                session_key=session_key,
                                description=desc,
                            ),
                            timeout=15,
                        )
                        if result and getattr(result, 'success', False):
                            sent = True
                            logger.info(
                                "[approval-delegation] Sent interactive approval card to admin %s:%s",
                                admin_platform, admin_chat,
                            )
                    except Exception as e:
                        logger.debug(
                            "[approval-delegation] Card send failed, falling back to text: %s", e,
                        )

                # Fallback: plain text message
                if not sent:
                    admin_msg = _tr(
                        "approval_request",
                        "🔐 Approval Delegation — Dangerous command requires admin approval\nUser: {user} (from {platform})\nReason: {desc}\n```\n{cmd}\n```\nReply /approve to approve | /deny to reject",
                        user=src_user_name,
                        platform=src_platform,
                        desc=desc,
                        cmd=cmd_preview,
                    )
                    sent = _send_message_from_sync(admin_adapter, admin_chat, admin_msg)

                if sent:
                    successful_admins += 1
                    logger.info(
                        "[approval-delegation] Sent approval request to admin %s:%s",
                        admin_platform, admin_chat,
                    )
                else:
                    logger.error(
                        "[approval-delegation] Failed to notify admin %s:%s",
                        admin_platform, admin_chat,
                    )
                    # Roll back the registered delegation since admin can't see it
                    try:
                        from gateway.approval_delegation.delegation import (
                            clear_delegation,
                        )
                        clear_delegation(admin_platform, admin_chat)
                    except Exception:
                        pass
            
            # Fallback: if no admin could be notified, fall through to the
            # original callback so the user can at least see the prompt in
            # their own chat. Otherwise the request would silently hang
            # until approval timeout, leaving the user staring at the
            # "waiting for admin" message with no recourse.
            if successful_admins == 0:
                logger.warning(
                    "[approval-delegation] No admins reachable; falling back to "
                    "original callback for session %s",
                    session_key[:16] if session_key else "?",
                )
                # Resume typing on the user's chat (we paused it earlier)
                if user_adapter:
                    try:
                        user_adapter.resume_typing_for_chat(src_chat_id)
                    except Exception:
                        pass
                    # Tell the user we couldn't reach an admin
                    fallback_msg = _tr(
                        "delegation_fallback",
                        "⚠️ Could not reach admin. Approval request will continue in your session.",
                    )
                    _send_message_from_sync(
                        user_adapter, src_chat_id, fallback_msg, metadata=src_chat_meta
                    )
                callback(approval_data)
        
        # Register with the delegated callback
        _original_register(session_key, _delegated_callback)
    
    # Apply patch
    approval_module.register_gateway_notify = _patched_register_gateway_notify
    logger.info("[approval-delegation] Patched register_gateway_notify")


def _patch_approve_deny_handlers() -> None:
    """Patch /approve and /deny to resolve delegated approvals."""
    import gateway.run as run_module
    
    # Import delegation functions
    from gateway.approval_delegation.delegation import (
        resolve_delegation, clear_delegation,
    )
    from tools.approval import resolve_gateway_approval, has_blocking_approval
    
    # Store original handlers
    _original_approve = run_module.GatewayRunner._handle_approve_command
    _original_deny = run_module.GatewayRunner._handle_deny_command
    
    async def _patched_handle_approve_command(self, event):
        """Patched /approve that handles delegated approvals."""
        source = event.source
        
        _src_platform = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        _src_chat_id = str(source.chat_id or "")
        
        # Check if this is a delegated approval
        delegation = resolve_delegation(_src_platform, _src_chat_id)
        
        if delegation is not None:
            # This is an admin resolving a delegated approval
            target_sk = delegation["session_key"]
            
            # Clean stale delegations
            while delegation is not None and not has_blocking_approval(target_sk):
                clear_delegation(_src_platform, _src_chat_id)
                delegation = resolve_delegation(_src_platform, _src_chat_id)
            
            if delegation is None:
                return _tr("approval_expired", "This approval request has expired.")
            
            # Parse args safely
            args_str = event.get_command_args()
            args = args_str.strip().lower().split() if args_str else []
            resolve_all = "all" in args
            remaining = [a for a in args if a != "all"]
            
            if any(a in {"always", "permanent", "permanently"} for a in remaining):
                choice = "always"
            elif any(a in {"session", "ses"} for a in remaining):
                choice = "session"
            else:
                choice = "once"
            
            # Resolve the approval
            count = resolve_gateway_approval(target_sk, choice, resolve_all=resolve_all)
            clear_delegation(_src_platform, _src_chat_id)
            
            if not count:
                return _tr("approval_expired", "This approval request has expired.")
            
            # Notify the original user (cross-platform)
            user_platform = delegation.get("user_platform", "")
            user_chat_id = delegation.get("user_chat_id", "")
            user_chat_meta = delegation.get("user_chat_meta")
            
            if user_chat_id and user_platform:
                user_adapter = _get_adapter(self, user_platform)
                if user_adapter:
                    await _send_message_safe(
                        user_adapter,
                        user_chat_id,
                        _tr("admin_approved", "✅ Admin approved. Executing..."),
                        metadata=user_chat_meta,
                    )
            
            # Resume typing on admin's chat
            adapter = self.adapters.get(source.platform)
            if adapter:
                adapter.resume_typing_for_chat(source.chat_id)
            
            logger.info(
                "Admin %s approved delegated command for session %s (%s)",
                source.user_name or source.user_id, target_sk[:16], choice,
            )
            
            from agent.i18n import t
            plural = "plural" if count > 1 else "singular"
            return t(f"gateway.approve.{choice}_{plural}", count=count)
        
        # Not a delegated approval, use original handler
        return await _original_approve(self, event)
    
    async def _patched_handle_deny_command(self, event):
        """Patched /deny that handles delegated approvals."""
        source = event.source
        
        _src_platform = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        _src_chat_id = str(source.chat_id or "")
        
        # Check if this is a delegated approval
        delegation = resolve_delegation(_src_platform, _src_chat_id)
        
        if delegation is not None:
            # This is an admin denying a delegated approval
            target_sk = delegation["session_key"]
            
            # Clean stale delegations
            while delegation is not None and not has_blocking_approval(target_sk):
                clear_delegation(_src_platform, _src_chat_id)
                delegation = resolve_delegation(_src_platform, _src_chat_id)
            
            if delegation is None:
                return _tr("approval_expired", "This approval request has expired.")
            
            # Deny the approval
            count = resolve_gateway_approval(target_sk, "deny", resolve_all=False)
            clear_delegation(_src_platform, _src_chat_id)
            
            if not count:
                return _tr("approval_expired", "This approval request has expired.")
            
            # Notify the original user (cross-platform)
            user_platform = delegation.get("user_platform", "")
            user_chat_id = delegation.get("user_chat_id", "")
            user_chat_meta = delegation.get("user_chat_meta")
            
            if user_chat_id and user_platform:
                user_adapter = _get_adapter(self, user_platform)
                if user_adapter:
                    await _send_message_safe(
                        user_adapter,
                        user_chat_id,
                        _tr("admin_denied", "❌ Admin denied the operation."),
                        metadata=user_chat_meta,
                    )
            
            # Resume typing on admin's chat
            adapter = self.adapters.get(source.platform)
            if adapter:
                adapter.resume_typing_for_chat(source.chat_id)
            
            logger.info(
                "Admin %s denied delegated command for session %s",
                source.user_name or source.user_id, target_sk[:16],
            )
            
            return _tr("denied_request", "Approval request denied.")
        
        # Not a delegated approval, use original handler
        return await _original_deny(self, event)
    
    # Apply patches
    run_module.GatewayRunner._handle_approve_command = _patched_handle_approve_command
    run_module.GatewayRunner._handle_deny_command = _patched_handle_deny_command
    
    logger.info("[approval-delegation] Patched /approve and /deny handlers")
