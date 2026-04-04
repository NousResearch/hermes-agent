"""CLI entry point for the hermes-agent A2A adapter.

Loads environment variables from ``~/.hermes/.env``, configures logging,
and starts the A2A HTTP server using uvicorn (the standard ASGI server).

Unlike the ACP adapter (which uses stdio JSON-RPC), A2A is HTTP-based and
requires a real HTTP server. uvicorn is the correct choice here — it is
already a hermes dependency (used by the [rl] extra) and is the standard
server for Starlette/FastAPI apps produced by the a2a-sdk.

Usage::

    python -m a2a_adapter
    # or
    hermes a2a
    # or
    hermes-a2a

Configure via environment variables:
    A2A_HOST          — bind host (default: 0.0.0.0)
    A2A_PORT          — bind port (default: 9000)
    A2A_KEY           — optional Bearer token to protect the endpoint
    AGENT_NAME        — name shown in Agent Card
    AGENT_DESCRIPTION — description shown in Agent Card
    AGENT_SKILLS      — comma-separated skill names
    AGENT_MODEL       — model name shown in Agent Card metadata
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from hermes_constants import get_hermes_home


def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _load_env() -> None:
    from hermes_cli.env_loader import load_hermes_dotenv

    hermes_home = get_hermes_home()
    loaded = load_hermes_dotenv(hermes_home=hermes_home)
    logger = logging.getLogger(__name__)
    if loaded:
        for env_file in loaded:
            logger.info("Loaded env from %s", env_file)
    else:
        logger.info("No .env found at %s, using system env", hermes_home / ".env")


def main() -> None:
    """Entry point: load env, configure logging, run the A2A server."""
    _setup_logging()
    _load_env()

    logger = logging.getLogger(__name__)

    # Ensure the project root is on sys.path so ``from run_agent import AIAgent`` works
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    host = os.getenv("A2A_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_PORT", "9000"))

    logger.info("Starting hermes-agent A2A adapter on http://%s:%d", host, port)
    logger.info("Agent Card: http://%s:%d/.well-known/agent.json", host, port)

    try:
        import uvicorn
        from .server import build_app

        app = build_app(port=port)
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except ImportError as e:
        logger.error(
            "Missing dependency: %s\n"
            "Install with: pip install 'hermes-agent[a2a]'",
            e,
        )
        sys.exit(1)
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt)")


if __name__ == "__main__":
    main()
