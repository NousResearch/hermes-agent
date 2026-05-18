"""CMH subprocess wrapper foundation.

Local foundation only. This package does not activate live routing,
Telegram sends, gateway behavior, cron, daemon, MCP, deploy, merge, or
production mutation.
"""

from agent.cmh_subprocess.budget_display import format_budget_status
from agent.cmh_subprocess.envelope import check_budget, increment_usage, load_envelope_state
from agent.cmh_subprocess.halt_flags import is_halted, load_halt_flags
from agent.cmh_subprocess.wrappers import prepare_claude_print_invocation, prepare_codex_print_invocation

__all__ = [
    "check_budget",
    "format_budget_status",
    "increment_usage",
    "is_halted",
    "load_envelope_state",
    "load_halt_flags",
    "prepare_claude_print_invocation",
    "prepare_codex_print_invocation",
]
