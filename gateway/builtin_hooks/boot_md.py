"""Built-in boot-md hook — run ~/.hermes/BOOT.md on gateway startup.

This hook is always registered. It silently skips if no BOOT.md exists.
To activate, create ``~/.hermes/BOOT.md`` with instructions for the
agent to execute on every gateway restart.

Example BOOT.md::

    # Startup Checklist

    1. Check if any cron jobs failed overnight
    2. Send a status update to Discord #general
    3. If there are errors in /opt/app/deploy.log, summarize them

The agent runs in a background thread so it doesn't block gateway
startup. If nothing needs attention, it replies with [SILENT] to
suppress delivery.
"""

import logging
import threading

logger = logging.getLogger("hooks.boot-md")

from hermes_constants import get_hermes_home
HERMES_HOME = get_hermes_home()
BOOT_FILE = HERMES_HOME / "BOOT.md"


def _build_boot_prompt(content: str) -> str:
    """Wrap BOOT.md content in a system-level instruction."""
    return (
        "You are running a startup boot checklist. Follow the BOOT.md "
        "instructions below exactly.\n\n"
        "---\n"
        f"{content}\n"
        "---\n\n"
        "Execute each instruction. If you need to send a message to a "
        "platform, use the send_message tool.\n"
        "If nothing needs attention and there is nothing to report, "
        "reply with ONLY: [SILENT]"
    )


def _run_boot_agent(content: str, model: str = "", runtime_kwargs: dict | None = None) -> None:
    """Spawn a one-shot agent session to execute the boot instructions.

    Args:
        content: BOOT.md file content.
        model: Resolved model name from gateway config (e.g. "glm-5.1").
        runtime_kwargs: Resolved provider credentials dict from the gateway
            runtime. Contains api_key, base_url, provider, api_mode, command,
            args, credential_pool.
    """
    try:
        from run_agent import AIAgent

        prompt = _build_boot_prompt(content)
        runtime_kwargs = runtime_kwargs or {}
        agent = AIAgent(
            model=model,
            **runtime_kwargs,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=20,
        )
        result = agent.run_conversation(prompt)
        response = result.get("final_response", "")
        if response and "[SILENT]" not in response:
            logger.info("boot-md completed: %s", response[:200])
        else:
            logger.info("boot-md completed (nothing to report)")
    except Exception as e:
        logger.error("boot-md agent failed: %s", e)


async def handle(event_type: str, context: dict) -> None:
    """Gateway startup handler — run BOOT.md if it exists."""
    if not BOOT_FILE.exists():
        return

    content = BOOT_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return

    logger.info("Running BOOT.md (%d chars)", len(content))

    # Extract model + runtime from gateway startup context.
    # Falls back to empty values for backward compat (e.g. if an older
    # gateway version emits gateway:startup without these fields).
    model = context.get("model", "") if isinstance(context, dict) else ""
    runtime_kwargs = context.get("runtime_kwargs") if isinstance(context, dict) else None

    # Run in a background thread so we don't block gateway startup.
    thread = threading.Thread(
        target=_run_boot_agent,
        args=(content, model, runtime_kwargs),
        name="boot-md",
        daemon=True,
    )
    thread.start()
