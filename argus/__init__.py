"""Argus - Agent Resource Guardian & Unified Supervisor.

Entropy detection and session monitoring for Hermes agents.
"""

# Core
from .argus import Argus

# Detection functions
from .entropy import (
    detect_repeat_tool_calls,
    detect_repeat_commands,
    detect_stuck_loops,
    detect_no_file_changes,
    detect_error_cascade,
    detect_budget_pressure,
)

# Actions
from .actions import (
    kill_session,
    restart_session,
    inject_prompt,
    strip_session_prefix,
)

# Notifications
from .notifications import (
    send_notification,
    send_telegram,
    send_discord,
    send_slack,
    send_matrix,
    send_webhook,
)

__all__ = [
    # Core
    "Argus",
    # Detection
    "detect_repeat_tool_calls",
    "detect_repeat_commands",
    "detect_stuck_loops",
    "detect_no_file_changes",
    "detect_error_cascade",
    "detect_budget_pressure",
    # Actions
    "kill_session",
    "restart_session",
    "inject_prompt",
    "strip_session_prefix",
    # Notifications
    "send_notification",
    "send_telegram",
    "send_discord",
    "send_slack",
    "send_matrix",
    "send_webhook",
]
