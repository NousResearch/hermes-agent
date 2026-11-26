"""
NOX V3 Plugin - Neural Operational eXpression
Combines token optimization with verification for balanced performance.
"""

from pathlib import Path
from typing import Dict, Any

# Plugin metadata
PLUGIN_NAME = "nox_v3"
PLUGIN_VERSION = "3.0.0"

# Default configuration
DEFAULT_CONFIG = {
    "enabled": False,
    "mode": "balanced",  # conservative | balanced | aggressive
    "max_daily_tokens": 10000,
    "latency_budget_ms": 50,
    "fast_path_threshold": 100,  # tokens - below this, skip verification
}

# Global state
_global_state = {
    "daily_token_usage": 0,
    "session_count": 0,
    "verification_success_rate": 1.0,
    "last_reset": None,
}


def get_config() -> Dict[str, Any]:
    """Get current NOX V3 configuration."""
    return DEFAULT_CONFIG.copy()


def get_state() -> Dict[str, Any]:
    """Get current NOX V3 state."""
    return _global_state.copy()


def update_state(updates: Dict[str, Any]) -> None:
    """Update NOX V3 state."""
    _global_state.update(updates)


def reset_daily_usage() -> None:
    """Reset daily token usage (call at midnight)."""
    _global_state["daily_token_usage"] = 0
    _global_state["last_reset"] = None


def is_enabled() -> bool:
    """Check if NOX V3 is enabled."""
    return get_config().get("enabled", False)


def get_mode() -> str:
    """Get current compression mode."""
    return get_config().get("mode", "balanced")


def get_latency_budget() -> int:
    """Get latency budget in milliseconds."""
    return get_config().get("latency_budget_ms", 50)


def get_fast_path_threshold() -> int:
    """Get fast path threshold in tokens."""
    return get_config().get("fast_path_threshold", 100)


def can_use_tokens(tokens_needed: int) -> bool:
    """Check if we have token budget available."""
    max_tokens = get_config().get("max_daily_tokens", 10000)
    current_usage = _global_state.get("daily_token_usage", 0)
    return (current_usage + tokens_needed) <= max_tokens


def record_token_usage(tokens: int) -> None:
    """Record token usage."""
    _global_state["daily_token_usage"] += tokens


def increment_session_count() -> None:
    """Increment active session count."""
    _global_state["session_count"] += 1


def decrement_session_count() -> None:
    """Decrement active session count."""
    _global_state["session_count"] = max(0, _global_state["session_count"] - 1)


def record_verification_result(success: bool) -> None:
    """Record verification result for success rate tracking."""
    total = _global_state.get("verification_count", 0) + 1
    successes = _global_state.get("verification_successes", 0) + (1 if success else 0)
    _global_state["verification_count"] = total
    _global_state["verification_successes"] = successes
    _global_state["verification_success_rate"] = successes / total if total > 0 else 1.0


# Plugin initialization
def plugin_init(ctx):
    """Initialize NOX V3 plugin."""
    ctx.log(f"NOX V3 v{PLUGIN_VERSION} initialized")
    ctx.log(f"Status: {'enabled' if is_enabled() else 'disabled'}")
    ctx.log(f"Mode: {get_mode()}")
    ctx.log(f"Latency budget: {get_latency_budget()}ms")
    ctx.log(f"Max daily tokens: {get_config().get('max_daily_tokens', 10000)}")


def plugin_cleanup(ctx):
    """Cleanup NOX V3 plugin."""
    ctx.log(f"NOX V3 cleanup - daily usage: {_global_state['daily_token_usage']} tokens")
    ctx.log(f"Sessions served: {_global_state['session_count']}")
    ctx.log(f"Verification success rate: {_global_state['verification_success_rate']:.2%}")


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register NOX V3 plugin hooks and commands."""
    from . import hooks, commands

    # Register hooks
    ctx.register_hook("pre_llm_call", hooks.pre_llm_call)
    ctx.register_hook("post_llm_call", hooks.post_llm_call)

    # Register slash command
    ctx.register_command(
        "nox",
        handler=commands.handle_nox_command,
        description="NOX V3 management - status, enable, disable, config",
    )
