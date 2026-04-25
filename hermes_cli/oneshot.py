"""Oneshot (-z) mode: send a prompt, get the final content block, exit.

Bypasses cli.py entirely.  No banner, no spinner, no session_id line,
no stderr chatter.  Just the agent's final text to stdout.

Toolsets = whatever the user has configured for "cli" in `hermes tools`.
Rules / memory / AGENTS.md / preloaded skills = same as a normal chat turn.
Approvals = auto-bypassed (HERMES_YOLO_MODE=1 is set for the call).
Working directory = the user's CWD (AGENTS.md etc. resolve from there as usual).
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import redirect_stderr, redirect_stdout


def run_oneshot(prompt: str) -> int:
    """Execute a single prompt and print only the final content block.

    Returns the exit code.  Caller should sys.exit() with the return.
    """
    # Silence every stdlib logger for the duration.  AIAgent, tools, and
    # provider adapters all log to stderr through the root logger; file
    # handlers added by setup_logging() keep working (they're attached to
    # the root logger's handler list, not affected by level), but no
    # bytes reach the terminal.
    logging.disable(logging.CRITICAL)

    # Auto-approve any shell / tool approvals.  Non-interactive by
    # definition — a prompt would hang forever.
    os.environ["HERMES_YOLO_MODE"] = "1"
    os.environ["HERMES_ACCEPT_HOOKS"] = "1"

    # Redirect stderr AND stdout to devnull for the entire call tree.
    # We'll print the final response to the real stdout at the end.
    real_stdout = sys.stdout
    devnull = open(os.devnull, "w")

    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            response = _run_agent(prompt)
    finally:
        try:
            devnull.close()
        except Exception:
            pass

    if response:
        real_stdout.write(response)
        if not response.endswith("\n"):
            real_stdout.write("\n")
        real_stdout.flush()
    return 0


def _run_agent(prompt: str) -> str:
    """Build an AIAgent exactly like a normal CLI chat turn would, then
    run a single conversation.  Returns the final response string."""
    # Imports are local so they don't run when hermes is invoked for
    # other commands (keeps top-level CLI startup cheap).
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.tools_config import _get_platform_tools
    from run_agent import AIAgent

    cfg = load_config()
    runtime = resolve_runtime_provider()

    # Resolve the model the user has configured for normal chat use.
    model_cfg = cfg.get("model") or {}
    if isinstance(model_cfg, str):
        model = model_cfg
    else:
        model = model_cfg.get("default") or model_cfg.get("model") or ""

    # Pull in whatever toolsets the user has enabled for "cli".
    # sorted() gives stable ordering; set→list for AIAgent's signature.
    toolsets_list = sorted(_get_platform_tools(cfg, "cli"))

    agent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        model=model,
        enabled_toolsets=toolsets_list,
        quiet_mode=True,
        platform="cli",
        credential_pool=runtime.get("credential_pool"),
        # Interactive callbacks are intentionally NOT wired beyond this
        # one.  In oneshot mode there's no user sitting at a terminal:
        #   - clarify  → returns a synthetic "pick a default" instruction
        #                so the agent continues instead of stalling on
        #                the tool's built-in "not available" error
        #   - sudo password prompt → terminal_tool gates on
        #                HERMES_INTERACTIVE which we never set
        #   - shell-hook approval → auto-approved via HERMES_ACCEPT_HOOKS=1
        #                (set above); also falls back to deny on non-tty
        #   - dangerous-command approval → bypassed via HERMES_YOLO_MODE=1
        #   - skill secret capture → returns gracefully when no callback set
        clarify_callback=_oneshot_clarify_callback,
    )

    # Belt-and-braces: make sure AIAgent doesn't invoke any streaming
    # display callbacks that would bypass our stdout capture.
    agent.suppress_status_output = True
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None

    return agent.chat(prompt) or ""


def _oneshot_clarify_callback(question: str, choices=None) -> str:
    """Clarify is disabled in oneshot mode — tell the agent to pick a
    default and proceed instead of stalling or erroring."""
    if choices:
        return (
            f"[oneshot mode: no user available. Pick the best option from "
            f"{choices} using your own judgment and continue.]"
        )
    return (
        "[oneshot mode: no user available. Make the most reasonable "
        "assumption you can and continue.]"
    )
