"""
Router integration hook for AIAgent.
Provides model override decisions for tool calls based on layer routing.
"""

import logging
from typing import Any, Optional, Dict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def should_override_model_for_tool(tool_name: str, current_model: str = None) -> Optional[str]:
    """
    Check if router should override the default model for this tool.
    Returns the override model name, or None to use default.
    
    Args:
        tool_name: Name of the tool being called
        current_model: Current model being used (optional, for fallback chain)
        
    Returns:
        Model name to use, or None to use default
    """
    if not tool_name or not isinstance(tool_name, str):
        return None
    if tool_name.startswith("_") or tool_name == "unknown":
        return None

    try:
        # Attempt to load router configuration
        router = get_router()
        if router is None:
            return None
            
        decision = router.route(tool_name, classification="tool_call")
        
        # Only override if model differs from current model
        if decision.model and decision.model != current_model:
            logger.info(
                f"🔀 Router override: {tool_name} → {decision.model} ({getattr(decision, 'reason', 'dynamic_route')})"
            )
            return decision.model

    except Exception as e:
        logger.debug(f"Router query for {tool_name} failed gracefully: {e}")
        return None

    return None


def get_router():
    """
    Get or initialize the router instance.
    Handles missing router module gracefully.
    """
    try:
        from router.three_tiers import get_router as _get_router
        return _get_router()
    except (ImportError, ModuleNotFoundError):
        logger.debug("Router module not available; routing disabled")
        return None
    except Exception as e:
        logger.debug(f"Failed to initialize router: {e}")
        return None


def get_router_stats() -> Dict[str, Any]:
    """Get routing statistics for monitoring."""
    try:
        router = get_router()
        if router is None:
            return {}
        if hasattr(router, 'get_stats'):
            return router.get_stats()
        return {}
    except Exception as e:
        logger.debug(f"Router stats unavailable: {e}")
        return {}


def is_routing_enabled() -> bool:
    """Check if layer routing is enabled in config."""
    try:
        from hermes_constants import get_hermes_home
        config_path = Path(get_hermes_home()) / "config.yaml"
        
        if not config_path.exists():
            return False
            
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
            
        layer_routing = config.get('layer_routing', {})
        return layer_routing.get('enabled', False)
    except Exception as e:
        logger.debug(f"Could not check routing config: {e}")
        return False


def get_primary_tool_name(messages: list) -> Optional[str]:
    """Extract the primary tool being called from the message history.
    
    Searches backwards through messages for the most recent assistant
    message with tool_calls, returning the first tool name found.
    
    Args:
        messages: Conversation message history
        
    Returns:
        Name of primary tool being called, or None if no tool calls found
    """
    if not messages:
        return None
        
    # Search backwards for assistant message with tool_calls
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                return tool_calls[0].get("function", {}).get("name")
    
    return None


def should_override_model_for_request(
    messages: list,
    current_model: str = None,
    api_kwargs: Dict[str, Any] = None,
) -> Optional[str]:
    """Determine if model should be overridden for the pending API request.
    
    This is the integration point called before performing the API call.
    It extracts tool information from messages and queries the router.
    
    Args:
        messages: Conversation message history
        current_model: Current model name
        api_kwargs: Current API request kwargs (for inspection)
        
    Returns:
        Override model name, or None to use current model
    """
    # Extract primary tool from messages
    tool_name = get_primary_tool_name(messages)
    
    if tool_name:
        return should_override_model_for_tool(tool_name, current_model)
    
    return None
