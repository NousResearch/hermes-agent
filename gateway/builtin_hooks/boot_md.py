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


def _run_boot_agent(content: str) -> None:
    """Spawn a one-shot agent session to execute the boot instructions."""
    try:
        from run_agent import AIAgent
        from hermes_cli.config import load_config

        # Inherit model/provider/api_mode from config.yaml so the boot
        # agent hits the user's configured endpoint. Without these,
        # AIAgent falls back to defaults (model="", api_mode=None)
        # and the init chain selects api_mode="chat_completions" — on
        # an openai-codex setup that POSTs to
        # chatgpt.com/backend-api/codex/chat/completions, a route the
        # Codex backend doesn't expose (only /responses exists) → 404.
        cfg = load_config()
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, dict):
            agent_kwargs = {
                "model": model_cfg.get("default") or "",
                "provider": model_cfg.get("provider") or None,
                "api_mode": model_cfg.get("api_mode") or None,
            }
        else:
            agent_kwargs = {"model": str(model_cfg or "")}

        prompt = _build_boot_prompt(content)
        agent = AIAgent(
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=20,
            **agent_kwargs,
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

    # Run in a background thread so we don't block gateway startup.
    thread = threading.Thread(
        target=_run_boot_agent,
        args=(content,),
        name="boot-md",
        daemon=True,
    )
    thread.start()
