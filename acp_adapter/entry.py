"""CLI entry point for the Hermes ACP agent server.

Launched by editors (Zed, JetBrains, etc.) or directly via ``hermes-acp``.
Communicates over stdio using JSON-RPC 2.0 as defined by the Agent Client
Protocol.

All logging is redirected to stderr so stdout remains clean for JSON-RPC.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure project root is importable (covers editable installs and dev usage).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_env() -> None:
    """Load ~/.hermes/.env via dotenv, mirroring hermes_cli.main behaviour."""
    try:
        from dotenv import load_dotenv
        from hermes_cli.config import get_env_path

        env_path = get_env_path()
        if env_path.exists():
            try:
                load_dotenv(dotenv_path=env_path, encoding="utf-8")
            except UnicodeDecodeError:
                load_dotenv(dotenv_path=env_path, encoding="latin-1")
    except ImportError:
        # python-dotenv or hermes_cli not available — rely on shell env.
        pass

    # Also try project-root .env as dev fallback.
    try:
        from dotenv import load_dotenv as _ld

        project_env = _PROJECT_ROOT / ".env"
        if project_env.exists():
            _ld(dotenv_path=project_env, override=False)
    except ImportError:
        pass


def _setup_logging() -> None:
    """Route all logging to stderr so stdout stays JSON-RPC only."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


async def run_acp_agent() -> None:
    """Create and run the Hermes ACP agent."""
    from acp import run_agent

    from acp_adapter.server import HermesACPAgent

    agent = HermesACPAgent()
    try:
        await run_agent(agent, use_unstable_protocol=True)
    finally:
        agent.shutdown()


def main() -> None:
    """Entry point for the ``hermes-acp`` console script."""
    _load_env()
    _setup_logging()
    asyncio.run(run_acp_agent())


if __name__ == "__main__":
    main()
