"""
NOX V3 Slash Commands - /nox status, enable, disable, config
"""

from typing import Dict, Any
from . import (
    get_config,
    get_state,
    update_state,
    is_enabled,
    get_mode,
    get_latency_budget,
    get_fast_path_threshold,
    reset_daily_usage,
    DEFAULT_CONFIG,
)


def format_status() -> str:
    """Format status message."""
    config = get_config()
    state = get_state()

    status = "✅ ENABLED" if config["enabled"] else "❌ DISABLED"
    mode = config["mode"].upper()
    latency = config["latency_budget_ms"]
    max_tokens = config["max_daily_tokens"]
    used_tokens = state["daily_token_usage"]
    token_percent = (used_tokens / max_tokens * 100) if max_tokens > 0 else 0
    sessions = state["session_count"]
    success_rate = state["verification_success_rate"]

    return f"""
╔════════════════════════════════════════════════════════════╗
║                    NOX V3 Status                           ║
╠════════════════════════════════════════════════════════════╣
║  Status: {status:<50} ║
║  Mode: {mode:<52} ║
║  Latency Budget: {latency}ms{' '*40} ║
╠════════════════════════════════════════════════════════════╣
║  Token Usage:                                            ║
║    Used: {used_tokens:>8} / {max_tokens:<8} ({token_percent:>5.1%}){' '*18} ║
║  Sessions: {sessions:>8}{' '*46} ║
║  Verification Success Rate: {success_rate:>6.1%}{' '*30} ║
╚════════════════════════════════════════════════════════════╝
"""


def format_config() -> str:
    """Format configuration message."""
    config = get_config()

    return f"""
╔════════════════════════════════════════════════════════════╗
║                  NOX V3 Configuration                      ║
╠════════════════════════════════════════════════════════════╣
║  enabled: {str(config['enabled']):<50} ║
║  mode: {config['mode']:<52} ║
║  max_daily_tokens: {config['max_daily_tokens']:<46} ║
║  latency_budget_ms: {config['latency_budget_ms']:<44} ║
║  fast_path_threshold: {config['fast_path_threshold']:<42} ║
╚════════════════════════════════════════════════════════════╝
"""


def handle_nox_command(raw_args: str) -> str:
    """
    Handle /nox slash command.

    Args:
        raw_args: Raw command arguments as a string

    Returns:
        Response message
    """
    # Parse arguments
    args = raw_args.strip().split() if raw_args else []
    config = get_config()

    if not args:
        return format_status()

    subcommand = args[0].lower()

    # /nox status
    if subcommand == "status":
        return format_status()

    # /nox enable
    elif subcommand == "enable":
        # Parse optional arguments
        mode = "balanced"
        max_tokens = None
        latency = None

        i = 1
        while i < len(args):
            if args[i] == "--mode" and i + 1 < len(args):
                mode = args[i + 1].lower()
                if mode not in ["conservative", "balanced", "aggressive"]:
                    return f"❌ Invalid mode: {mode}. Use: conservative, balanced, or aggressive"
                i += 2
            elif args[i] == "--max-tokens" and i + 1 < len(args):
                try:
                    max_tokens = int(args[i + 1])
                except ValueError:
                    return f"❌ Invalid max-tokens: {args[i + 1]}"
                i += 2
            elif args[i] == "--latency" and i + 1 < len(args):
                try:
                    latency = int(args[i + 1])
                except ValueError:
                    return f"❌ Invalid latency: {args[i + 1]}"
                i += 2
            else:
                i += 1

        # Update configuration
        updates = {"enabled": True, "mode": mode}
        if max_tokens is not None:
            updates["max_daily_tokens"] = max_tokens
        if latency is not None:
            updates["latency_budget_ms"] = latency

        # Update both local config and plugin config
        for key, value in updates.items():
            DEFAULT_CONFIG[key] = value
            config[key] = value

        return f"""
✅ NOX V3 ENABLED

Mode: {mode.upper()}
Max Daily Tokens: {updates.get('max_daily_tokens', config.get('max_daily_tokens', 10000))}
Latency Budget: {updates.get('latency_budget_ms', config.get('latency_budget_ms', 50))}ms

NOX will now automatically apply to all LLM calls in all sessions.
Use '/nox disable' to stop.
Use '/nox status' to check usage.
"""

    # /nox disable
    elif subcommand == "disable":
        # Update configuration
        DEFAULT_CONFIG["enabled"] = False
        config["enabled"] = False

        state = get_state()
        sessions = state["session_count"]

        return f"""
❌ NOX V3 DISABLED

NOX will stop applying to new sessions.
Existing sessions will continue until complete.

Sessions served: {sessions}
Daily token usage: {state['daily_token_usage']}
"""

    # /nox config
    elif subcommand == "config":
        # Parse optional arguments
        updates = {}

        i = 1
        while i < len(args):
            if args[i] == "--mode" and i + 1 < len(args):
                mode = args[i + 1].lower()
                if mode not in ["conservative", "balanced", "aggressive"]:
                    return f"❌ Invalid mode: {mode}. Use: conservative, balanced, or aggressive"
                updates["mode"] = mode
                i += 2
            elif args[i] == "--max-tokens" and i + 1 < len(args):
                try:
                    updates["max_daily_tokens"] = int(args[i + 1])
                except ValueError:
                    return f"❌ Invalid max-tokens: {args[i + 1]}"
                i += 2
            elif args[i] == "--latency" and i + 1 < len(args):
                try:
                    updates["latency_budget_ms"] = int(args[i + 1])
                except ValueError:
                    return f"❌ Invalid latency: {args[i + 1]}"
                i += 2
            elif args[i] == "--fast-path" and i + 1 < len(args):
                try:
                    updates["fast_path_threshold"] = int(args[i + 1])
                except ValueError:
                    return f"❌ Invalid fast-path threshold: {args[i + 1]}"
                i += 2
            else:
                i += 1

        # Apply updates
        for key, value in updates.items():
            DEFAULT_CONFIG[key] = value
            config[key] = value

        if updates:
            return f"""
✅ NOX V3 Configuration Updated

{format_config()}
"""
        else:
            return format_config()

    # /nox reset
    elif subcommand == "reset":
        reset_daily_usage()

        return "✅ NOX V3 daily token usage has been reset."

    # Unknown subcommand
    else:
        return f"""
❌ Unknown subcommand: {subcommand}

Available commands:
  /nox status     - Show current status and usage
  /nox enable     - Enable NOX (optional: --mode, --max-tokens, --latency)
  /nox disable    - Disable NOX
  /nox config     - Show/update configuration
  /nox reset      - Reset daily token usage

Examples:
  /nox enable --mode balanced --max-tokens 20000
  /nox config --mode aggressive --latency 100
"""
