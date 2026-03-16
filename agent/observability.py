"""Shared observability utilities for Langfuse integration.

This module provides a centralized import point for Langfuse decorators
to ensure graceful degradation when Langfuse is not installed.
"""

import functools
import logging
import os
import threading
import atexit
from contextvars import ContextVar
from typing import Any, Callable, Optional, ParamSpec, TypeVar, cast


P = ParamSpec("P")
R = TypeVar("R")


_langfuse_enabled_override: ContextVar[bool | None] = ContextVar(
    "langfuse_enabled_override",
    default=None,
)

# Langfuse observability (optional - gracefully degrades if not installed)
try:
    from langfuse import Langfuse, observe as _langfuse_observe
    LANGFUSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    _langfuse_observe = None


def _is_langfuse_enabled() -> bool:
    """Check if Langfuse is enabled via config or env var.

    Priority:
    1. ContextVar override (set by AIAgent/tool runtime)
    2. HERMES_LANGFUSE_ENABLED env var (set by Hermes config)
    3. LANGFUSE_ENABLED env var (direct user override)
    4. Default to False unless explicitly enabled
    """
    override = _langfuse_enabled_override.get()
    if override is not None:
        return bool(override)

    hermes_enabled = os.getenv("HERMES_LANGFUSE_ENABLED", "").lower()
    if hermes_enabled in ("true", "1", "yes"):
        return True
    if hermes_enabled in ("false", "0", "no"):
        return False

    langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "").lower()
    if langfuse_enabled in ("true", "1", "yes"):
        return True
    if langfuse_enabled in ("false", "0", "no"):
        return False

    return False


def observe(
    name: str | None = None,
    as_type: Any = None,
    **kwargs: Any,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Langfuse observe decorator that respects runtime config.

    We check enablement at call time (env may be bridged after imports), but we
    must not re-apply the Langfuse decorator on every function invocation.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if not LANGFUSE_AVAILABLE or _langfuse_observe is None:
            return func

        # Capture as local for type-checkers (and to avoid global mutation surprises).
        langfuse_observe = _langfuse_observe

        decorated_func: Optional[Callable[P, R]] = None
        decorate_failed = False
        decorate_lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*f_args: P.args, **f_kwargs: P.kwargs) -> R:
            nonlocal decorated_func, decorate_failed

            if not _is_langfuse_enabled():
                return func(*f_args, **f_kwargs)

            if decorate_failed:
                return func(*f_args, **f_kwargs)

            if decorated_func is None:
                with decorate_lock:
                    if decorated_func is None and not decorate_failed:
                        try:
                            decorated = langfuse_observe(
                                name=name,
                                as_type=as_type,
                                **kwargs,
                            )(func)
                            decorated_func = cast(Callable[P, R], decorated)
                        except Exception:
                            decorate_failed = True
                            logging.getLogger(__name__).warning(
                                "Langfuse observe decorator failed to initialize; proceeding without observation.",
                                exc_info=True,
                            )
                            return func(*f_args, **f_kwargs)

            if decorated_func is None:
                return func(*f_args, **f_kwargs)

            return decorated_func(*f_args, **f_kwargs)

        return wrapper

    return decorator


# Langfuse client singleton for thread safety
_langfuse_client_singleton = None
_langfuse_client_lock = threading.Lock()


def set_langfuse_enabled(enabled: bool | None) -> None:
    """Set a per-thread override for tool-level observability decorators."""
    _langfuse_enabled_override.set(enabled)


def close_langfuse_client() -> None:
    """Best-effort shutdown for the Langfuse client singleton."""
    global _langfuse_client_singleton
    with _langfuse_client_lock:
        client = _langfuse_client_singleton
        _langfuse_client_singleton = None
    if client is None:
        return
    for meth in ("flush", "shutdown", "close"):
        fn = getattr(client, meth, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def configure_langfuse_client(client) -> None:
    """Force-set the Langfuse client singleton.

    Used by short-lived subprocess runners to ensure all instrumentation paths
    (e.g. tool spans created in model_tools) share the same configured client.
    """
    global _langfuse_client_singleton
    with _langfuse_client_lock:
        _langfuse_client_singleton = client


# Ensure the singleton is cleaned up on process exit.
try:  # pragma: no cover
    atexit.register(close_langfuse_client)
except Exception:
    pass



def get_langfuse_client():
    """Get or create the Langfuse client singleton.
    
    Uses double-checked locking for thread safety.
    Returns None if Langfuse is not available.
    """
    global _langfuse_client_singleton
    
    if not LANGFUSE_AVAILABLE or Langfuse is None:
        return None
    
    if _langfuse_client_singleton is None:
        with _langfuse_client_lock:
            if _langfuse_client_singleton is None:
                _langfuse_client_singleton = Langfuse()
    return _langfuse_client_singleton


def validate_langfuse_sample_rate() -> float:
    """Validate Langfuse sample rate from environment variables.
    
    Returns validated sample rate between 0.0 and 1.0.
    Defaults to 1.0 if not set or invalid.
    Logs warnings for invalid values.
    
    This should be called at config load time to fail fast on bad config.
    """
    logger = logging.getLogger(__name__)
    # Backward-compatible precedence: explicit LANGFUSE_SAMPLE_RATE wins.
    env_value = os.getenv("LANGFUSE_SAMPLE_RATE")
    if env_value is None:
        env_value = os.getenv("HERMES_LANGFUSE_SAMPLE_RATE")
    
    if env_value is None:
        return 1.0
    
    try:
        sample_rate = float(env_value)
        if not (0.0 <= sample_rate <= 1.0):
            logger.warning(
                "Invalid Langfuse sample rate: %s (must be 0.0-1.0), using 1.0",
                sample_rate,
            )
            return 1.0
        return sample_rate
    except ValueError:
        logger.warning("Invalid Langfuse sample rate format %r, using 1.0", env_value)
        return 1.0
