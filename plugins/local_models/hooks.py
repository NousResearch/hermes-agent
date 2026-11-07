"""Plugin hooks for local-models.

Allows other plugins to extend local-models behavior without modifying core.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Registry of launch override callbacks
_launch_override_callbacks: List[Callable] = []


def register_launch_override(callback: Callable) -> None:
    """Register a callback to override launch parameters.
    
    Callback signature: callback(launch_params: dict, model_id: str) -> dict
    Should return modified launch_params.
    """
    _launch_override_callbacks.append(callback)
    logger.debug(f"Registered launch override callback from {callback.__module__}")


def apply_launch_overrides(launch_params: dict, model_id: Optional[str] = None) -> dict:
    """Apply all registered launch override callbacks.
    
    Each callback receives the current launch_params dict and can modify it.
    Callbacks are applied in registration order.
    """
    result = launch_params
    for callback in _launch_override_callbacks:
        try:
            result = callback(result, model_id)
        except Exception as e:
            logger.warning(f"Launch override callback {callback} failed: {e}")
    return result


# Registry of UI pane callbacks (for desktop plugin integration)
_ui_pane_callbacks: List[Callable] = []


def register_ui_pane(callback: Callable) -> None:
    """Register a UI pane to be displayed in the Local Models tab.
    
    Callback signature: callback() -> dict with {id, title, component}
    """
    _ui_pane_callbacks.append(callback)


def get_ui_panes() -> List[Dict[str, Any]]:
    """Get all registered UI panes."""
    panes = []
    for callback in _ui_pane_callbacks:
        try:
            pane = callback()
            if pane:
                panes.append(pane)
        except Exception as e:
            logger.warning(f"UI pane callback failed: {e}")
    return panes
