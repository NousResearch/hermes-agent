"""
Graceful Shutdown Manager for Hermes-Agent.

This module provides a singleton ShutdownManager that coordinates shutdown hooks
across the application. It replaces scattered atexit.register() calls with a
unified registry that supports:
- Priority-based ordering (lower number = runs first)
- SIGINT/SIGTERM signal handling
- Thread-safe registration and execution

Priority ordering:
    Priority 40: Credential files cleanup (symlinks, registry)
    Priority 50: Terminal sandbox cleanup
    Priority 60: Browser session cleanup

Usage:
    from agent.hermes.shutdown import ShutdownManager

    sm = ShutdownManager.get_instance()
    sm.register(my_cleanup_callback, priority=50)
    sm.execute()  # Run all hooks in priority order
"""

import threading
import atexit
import signal
import logging
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)


class ShutdownManager:
    """
    Singleton shutdown manager with priority-based hook ordering.

    Shutdown hooks are executed in ASCENDING priority order (lower number first).
    This allows dependent cleanup sequences like:
        40 -> credential cleanup (must happen first)
        50 -> terminal cleanup (depends on credentials being cleared)
        60 -> browser cleanup (depends on terminals being cleared)
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._hooks: List[Tuple[int, Callable]] = []  # (priority, callback)
        self._shutdown_in_progress = False
        self._registered = False
        self._local_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ShutdownManager":
        """Get the singleton ShutdownManager instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register(self, callback: Callable, priority: int = 100) -> None:
        """
        Register a shutdown hook.

        Args:
            callback: Function to call during shutdown
            priority: ASCENDING priority (lower numbers run FIRST).
                     Typical usage:
                       priority=40: credential cleanup (symlinks, registry)
                       priority=50: sandbox cleanup
                       priority=60: browser cleanup
        """
        with self._local_lock:
            self._hooks.append((priority, callback))
            self._hooks.sort(key=lambda x: x[0])
            logger.debug(f"Shutdown hook registered: priority={priority}")

    def execute(self) -> None:
        """
        Execute all shutdown hooks in priority order (ascending).

        Hooks are executed exactly once, even if execute() is called multiple times.
        Exceptions in individual hooks are logged but do not stop other hooks.
        """
        with self._local_lock:
            if self._shutdown_in_progress:
                return
            self._shutdown_in_progress = True
            hooks = list(self._hooks)

        logger.info(f"Executing {len(hooks)} shutdown hooks...")
        for priority, callback in hooks:
            try:
                logger.debug(f"Running shutdown hook: priority={priority}")
                callback()
            except Exception as e:
                logger.warning(f"Shutdown hook (priority={priority}) failed: {e}")

    def _setup_atexit_handler(self) -> None:
        """Register the atexit handler if not already registered."""
        if not self._registered:
            atexit.register(self.execute)
            self._registered = True

    def setup_signal_handlers(self, sigint_handler: Callable = None, sigterm_handler: Callable = None) -> None:
        """
        Set up SIGINT and SIGTERM signal handlers.

        Args:
            sigint_handler: Custom handler for SIGINT (default: execute hooks)
            sigterm_handler: Custom handler for SIGTERM (default: execute hooks)

        Note:
            Signal handlers can only be set from the main thread. On Python 3.13+
            or in non-main threads, this method logs a warning and skips handler
            registration rather than raising RuntimeError.
        """
        self._setup_atexit_handler()

        def default_sigint_handler(signum, frame):
            logger.info("SIGINT received, initiating graceful shutdown...")
            self.execute()

        def default_sigterm_handler(signum, frame):
            logger.info("SIGTERM received, initiating graceful shutdown...")
            self.execute()

        # signal.signal() only works in the main thread. In Python 3.13+ or in
        # embedded environments (tmux, IDE consoles), we may be in a non-main thread.
        if threading.current_thread() is not threading.main_thread():
            logger.warning(
                "ShutdownManager: signal handlers not registered — "
                "setup_signal_handlers called from non-main thread %s. "
                "Signal-based shutdown will not work in this environment.",
                threading.current_thread().name,
            )
            return

        try:
            signal.signal(signal.SIGINT, sigint_handler or default_sigint_handler)
            signal.signal(signal.SIGTERM, sigterm_handler or default_sigterm_handler)
            logger.debug("ShutdownManager: signal handlers registered")
        except (OSError, RuntimeError) as e:
            logger.warning(
                "ShutdownManager: could not register signal handlers: %s. "
                "Signal-based shutdown will not work in this environment.",
                e,
            )

    def clear_hooks(self) -> None:
        """Clear all registered hooks (mainly for testing)."""
        with self._local_lock:
            self._hooks.clear()
            self._shutdown_in_progress = False

    @property
    def hook_count(self) -> int:
        """Return the number of registered hooks."""
        with self._local_lock:
            return len(self._hooks)
