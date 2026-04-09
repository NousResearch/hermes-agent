"""Argus - Agent Resource Guardian & Unified Supervisor."""

from .actions import (
    inject_prompt,
    kill_session,
    restart_session,
    strip_session_prefix,
)
from .argus import Argus
from .daemon_mgmt import (
    argus_launchd_install,
    argus_launchd_status,
    argus_launchd_uninstall,
    generate_argus_launchd_plist,
    get_argus_launchd_label,
    get_argus_launchd_plist_path,
    get_argus_running_pid,
    is_argus_running,
    remove_argus_pid_file,
    write_argus_pid_file,
)
from .circuit_breaker import CircuitBreaker, check_circuits, format_circuit_event
from .cost_monitor import CostMonitor, check_costs, format_cost_alert
from .entropy import (
    detect_budget_pressure,
    detect_error_cascade,
    detect_no_file_changes,
    detect_repeat_commands,
    detect_repeat_tool_calls,
    detect_stuck_loops,
)
from .metrics import MetricsCollector, write_metrics_file
from .ml_data import (
    HolographicMemoryBridge,
    MLDataExporter,
    export_entropy_event,
)
from .notifications import (
    send_discord,
    send_matrix,
    send_notification,
    send_slack,
    send_telegram,
    send_via_gateway,
    send_webhook,
)
from .setup import main as run_setup
from .subprocess_utils import safe_subprocess
from .venv_utils import (
    build_argus_subprocess_env,
    detect_hermes_venv,
    get_hermes_python,
    get_venv_path,
    get_venv_python,
    is_running_in_venv,
    resolve_venv_python,
)

__all__ = [
    "Argus",
    "detect_budget_pressure",
    "detect_error_cascade",
    "detect_no_file_changes",
    "detect_repeat_commands",
    "detect_repeat_tool_calls",
    "detect_stuck_loops",
    "kill_session",
    "restart_session",
    "inject_prompt",
    "strip_session_prefix",
    "send_discord",
    "send_matrix",
    "send_notification",
    "send_slack",
    "send_telegram",
    "send_via_gateway",
    "send_webhook",
    "MetricsCollector",
    "write_metrics_file",
    "MLDataExporter",
    "HolographicMemoryBridge",
    "export_entropy_event",
    "CostMonitor",
    "check_costs",
    "format_cost_alert",
    "CircuitBreaker",
    "check_circuits",
    "format_circuit_event",
    "is_running_in_venv",
    "get_venv_path",
    "get_venv_python",
    "detect_hermes_venv",
    "get_hermes_python",
    "build_argus_subprocess_env",
    "resolve_venv_python",
    # Subprocess utilities
    "safe_subprocess",
    # Daemon management
    "write_argus_pid_file",
    "remove_argus_pid_file",
    "get_argus_running_pid",
    "is_argus_running",
    "get_argus_launchd_label",
    "get_argus_launchd_plist_path",
    "generate_argus_launchd_plist",
    "argus_launchd_install",
    "argus_launchd_uninstall",
    "argus_launchd_status",
    # Setup
    "run_setup",
]
