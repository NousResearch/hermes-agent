"""``model.lock_switching``: an operator setting that turns off model switching.

For installations where something outside Hermes chooses the model, such as a gateway that routes requests by
policy. :func:`hermes_cli.model_switch.switch_model` checks it first, so ``/model`` and every other surface that
switches through it (CLI, TUI, ACP, messaging gateway, dashboard) refuse with the operator's message through
their existing error path instead of switching.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SWITCH_LOCK_MESSAGE = "Model switching is turned off for this installation."


def model_switch_lock_message(cfg: Any = None) -> str:
    """The message to show when ``model.lock_switching`` is set, else ``""``.

    ``model.lock_switching_message`` replaces the default text, e.g. to say how the model is chosen. An
    unreadable config never locks."""
    if cfg is None:
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
        except Exception:
            logger.debug("model.lock_switching: config unreadable; switching stays unlocked", exc_info=True)
            return ""
    model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
    if not isinstance(model_cfg, dict) or not model_cfg.get("lock_switching"):
        return ""
    message = model_cfg.get("lock_switching_message")
    return message.strip() if isinstance(message, str) and message.strip() else DEFAULT_SWITCH_LOCK_MESSAGE
